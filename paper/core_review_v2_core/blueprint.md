# Executable blueprint: PAPER-PIPELINE-V2-CORE-01

Outline: `b96c343682e90ea7c7d581cbebe68de3d16b972c9f9355af8aa50f9d800c3dcd`
Snapshot: `4361f9c291c2e836f728bcd486b726a8f7dad1d407d0eed937589790ae077598`

## Resolution summary

- Outline nodes: 39
- Enabled nodes: 2
- Disabled nodes: 37
- Structural rule: no merge, split, rename, reorder, or silent omission.

## METHOD-P03 - Proposition 2: Vanishing Weighted Far-Field Gradient

- Parent outline block: `a28d858a65e1ca1c2d492ae622405d4816d9f956abe0b886a75298c957ba2654`
- Reader question: Why use the exponential form rather than an arbitrary taper?
- Paragraph claim: Under the registered finite-order score-growth assumption, the exponential remoteness envelope drives the weighted far-field contribution to zero without asserting that sample utility itself decays exponentially.
- Sentence plan:
  1. **motivation** — State that the exponential form is chosen for a tail-control property, not a utility model.
  2. **assumption** — Introduce the finite-order bound C(1+r)^k on the unweighted score-times-advantage norm.
  3. **analytic_result** — State that exp(-lambda r) dominates every finite polynomial order and makes the weighted term vanish.
  4. **claim_boundary** — Exclude universal taper rankings and exponential utility-decay interpretations.
- Evidence refs: `PROP-EXPONENTIAL-FAR-FIELD`, `PROOF-EXPONENTIAL-FAR-FIELD`
- Metric paths: none
- Figure refs: none
- Table refs: none
- Theorem/equation refs: `PROP-EXPONENTIAL-FAR-FIELD`, `EQ-EXPONENTIAL-REMOTE-WEIGHT`
- Reviewer objection: The exponential form may be an arbitrary utility-decay assumption.
- Response: The proof uses only finite-order score growth; it makes no assumption about utility decay.
- Allowed conclusions: exponential weighting gives a far-field tail guarantee under finite-order score growth
- Forbidden conclusions: exponential weighting is universally superior to every taper; sample utility decays exponentially with remoteness
- Transition: The empirical intervention then tests whether the controlled far-field path is the instability channel.

## EXP-P04 - RQ2b: Targeted Causal Transmission

- Parent outline block: `16bb56f504d2c47f264351b3f0d160ce052d0d75ad2d26ab00c985df098be976`
- Reader question: Do large far-field updates actually cause the observed instability?
- Paragraph claim: In the controlled C-U1 intervention, near-field removal does not rescue the policy, whereas deleting or capping the far-field path prevents fixed-variance task collapse and learnable-variance support contraction under the registered controls.
- Sentence plan:
  1. **setup** — Define the matched four-way intervention, paired held-out seeds, and separate event taxonomy.
  2. **fixed_variance_failure** — Report Baseline and Near-zero reward and task-collapse counts from exact snapshot metrics.
  3. **fixed_variance_rescue** — Report Far-zero and Far-cap reward, confidence intervals, and zero collapse counts.
  4. **budget_controls** — Report Global-scale and Far-to-near as diagnostic fixed-budget controls without turning them into a method ranking.
  5. **learnable_variance_audit** — Report support-boundary counts and onset for Baseline and Near-zero, plus zero events for Far-zero and Far-cap.
  6. **numerical_separation** — Report NaN/Inf separately and state that parameters remain finite.
  7. **calibrated_conclusion** — Conclude only that the far-field component is the dominant causal transmission path in this controlled environment.
- Evidence refs: `EVID-CU1-E3-FIXED`, `EVID-CU1-E3-LEARNABLE`, `AUDIT-CU1-E3-TERMINAL`
- Metric paths: `methods.baseline.fixed_variance.reward`, `methods.baseline.fixed_variance.task_collapse_count`, `methods.near_zero.fixed_variance.reward`, `methods.near_zero.fixed_variance.task_collapse_count`, `methods.far_zero.fixed_variance.reward`, `methods.far_zero.fixed_variance.reward_ci95`, `methods.far_zero.fixed_variance.task_collapse_count`, `methods.far_cap.fixed_variance.reward`, `methods.far_cap.fixed_variance.reward_ci95`, `methods.far_cap.fixed_variance.task_collapse_count`, `methods.global_scale.fixed_variance.reward`, `methods.far_to_near.fixed_variance.reward`, `methods.baseline.learnable_variance.support_boundary_count`, `methods.baseline.learnable_variance.support_onset_mean`, `methods.near_zero.learnable_variance.support_boundary_count`, `methods.near_zero.learnable_variance.support_onset_mean`, `methods.far_zero.learnable_variance.support_boundary_count`, `methods.far_cap.learnable_variance.support_boundary_count`, `methods.baseline.learnable_variance.nan_inf_count`, `methods.near_zero.learnable_variance.nan_inf_count`, `methods.far_zero.learnable_variance.nan_inf_count`, `methods.far_cap.learnable_variance.nan_inf_count`
- Figure refs: `cu1_e3_fixed_reward.pdf`
- Table refs: `cu1_e3_results.tex`
- Theorem/equation refs: none
- Reviewer objection: The rescue may reflect removing all negative information or merely changing the total update budget.
- Response: Near-zero removes local negative information without rescue; Far-cap retains bounded far influence, and the registered Global-scale and Far-to-near controls isolate budget effects.
- Allowed conclusions: the far-field component is the dominant causal transmission path in controlled C-U1; the learnable-variance event is support contraction rather than NaN/Inf failure
- Forbidden conclusions: C-U1 demonstrates OOD generalization; the intervention proves a universal cause of off-policy collapse; the controls establish a universal method ranking
- Transition: Later sections must test whether the same mechanism transfers to external tasks before broadening the claim.

## Disabled nodes

- `ABSTRACT-P01` — not_selected_for_core_vertical_slice
- `INTRO-P01` — not_selected_for_core_vertical_slice
- `INTRO-P02` — not_selected_for_core_vertical_slice
- `INTRO-P03` — not_selected_for_core_vertical_slice
- `INTRO-P04` — not_selected_for_core_vertical_slice
- `INTRO-P05` — not_selected_for_core_vertical_slice
- `INTRO-P06` — not_selected_for_core_vertical_slice
- `RELATED-P01` — not_selected_for_core_vertical_slice
- `RELATED-P02` — not_selected_for_core_vertical_slice
- `RELATED-P03` — not_selected_for_core_vertical_slice
- `SETUP-P01` — not_selected_for_core_vertical_slice
- `SETUP-P02` — not_selected_for_core_vertical_slice
- `THEORY-P01` — not_selected_for_core_vertical_slice
- `THEORY-P02` — not_selected_for_core_vertical_slice
- `THEORY-P03` — not_selected_for_core_vertical_slice
- `THEORY-P04` — not_selected_for_core_vertical_slice
- `THEORY-P05` — not_selected_for_core_vertical_slice
- `METHOD-P01` — not_selected_for_core_vertical_slice
- `METHOD-P02` — not_selected_for_core_vertical_slice
- `METHOD-P04` — not_selected_for_core_vertical_slice
- `EXP-P01` — not_selected_for_core_vertical_slice
- `EXP-P02` — not_selected_for_core_vertical_slice
- `EXP-P03` — not_selected_for_core_vertical_slice
- `EXP-P05` — not_selected_for_core_vertical_slice
- `EXP-P06` — not_selected_for_core_vertical_slice
- `DISC-P01` — not_selected_for_core_vertical_slice
- `DISC-P02` — not_selected_for_core_vertical_slice
- `DISC-P03` — not_selected_for_core_vertical_slice
- `APP-PROOF-P01` — not_selected_for_core_vertical_slice
- `APP-GAUSS-P01` — not_selected_for_core_vertical_slice
- `APP-CAT-P01` — not_selected_for_core_vertical_slice
- `APP-ENV-P01` — not_selected_for_core_vertical_slice
- `APP-PROT-P01` — not_selected_for_core_vertical_slice
- `APP-RES-P01` — not_selected_for_core_vertical_slice
- `APP-REPRO-P01` — not_selected_for_core_vertical_slice
- `APP-DRO-P01` — not_selected_for_core_vertical_slice
- `APP-CORR-P01` — not_selected_for_core_vertical_slice
