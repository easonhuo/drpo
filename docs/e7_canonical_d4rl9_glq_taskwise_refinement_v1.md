# E7 D4RL-9 task-wise GLQ refinement pilot

## Identity and scope

- Proposed experiment ID: `EXT-H-E7-D4RL9-GLQ-REFINE-01`
- Source experiment and canonical RunSpec lineage: `EXT-H-E7-BENCH-01`
- Scientific status: development refinement pilot only
- Parent run: `E7_D4RL9_GLQ_TASKWISE_TUNING_20260726_01`
- Parent result commit: `easonhuo/drpo-results@088b703c6df98e2fa5807d471260d3c7241c7614`
- Parent source commit: `a0e4be818cbd780ac6ac36e0a56fa44de89493bf`

This stage refines the task-specific controller coefficients selected by the first 540-branch coarse sweep. It does not rerun the first-round points. It adds five new candidates for each of Global, Reciprocal-Linear, and Reciprocal-Quadratic in each of the nine D4RL tasks, using the same four development seeds.

The resulting matrix is:

\[
9\ \text{tasks}\times 4\ \text{development seeds}\times
(5_G+5_L+5_Q)=540\ \text{branches}.
\]

The terminal audit binds to the delivered parent `TERMINAL_AUDIT.json` by SHA-256 and then selects from ten total candidates per task/method cell. This is still development-set parameter selection, not held-out confirmation and not a formal cross-method ranking.

## Frozen invariants

The following values are unchanged from round one:

- datasets and order: Hopper, Walker2d, and HalfCheetah medium / medium-replay / medium-expert;
- development seeds: `200,201,202,203`;
- untouched held-out seeds: `204,205,206,207`;
- canonical trainer and source fingerprints;
- `variant=iqlv_exp_rank`;
- canonical `alpha=0.11`, `tau=0.5`, `temp=5.0`;
- `1,000,000` updates, batch size `256`, learning rate `3e-4`;
- evaluation every `50,000` updates with ten episodes;
- late window: `750k,800k,850k,900k,950k,1M`;
- `R_ref=2.0`;
- maximum parallel workers: `60`.

Positive advantages remain unchanged. Global multiplies every negative advantage by `alpha × negative_scale`. Reciprocal-Linear and Reciprocal-Quadratic fix `negative_scale=1` and use

\[
w_L(u)=\alpha(1+c u)^{-1},\qquad
w_Q(u)=\alpha(1+c u^2)^{-1},\qquad
u=r/R_{\mathrm{ref}}.
\]

`R_ref` is not co-tuned with `c`.

## Task-specific second-round grids

| Dataset | Global new scales | Linear new `c` | Quadratic new `c` |
|---|---|---|---|
| hopper-medium | 0.012, 0.018, 0.024, 0.045, 0.065 | 0.6, 0.75, 1.25, 1.6, 2.2 | 0.05, 0.1, 0.2, 0.3, 0.4 |
| hopper-medium-replay | 0.15, 0.25, 0.4, 0.65, 1.0 | 0.6, 0.75, 1.25, 1.6, 2.2 | 0.05, 0.1, 0.2, 0.3, 0.4 |
| hopper-medium-expert | 0.0015, 0.002, 0.0045, 0.006, 0.008 | 15, 20, 25, 40, 60 | 0.6, 0.75, 1.25, 1.6, 2.2 |
| walker2d-medium | 0.0015, 0.002, 0.0045, 0.006, 0.008 | 1.5, 2, 4.5, 6.5, 8 | 0.05, 0.1, 0.2, 0.3, 0.4 |
| walker2d-medium-replay | 0.0015, 0.002, 0.0045, 0.006, 0.008 | 0.05, 0.1, 0.2, 0.3, 0.4 | 45, 60, 90, 150, 300 |
| walker2d-medium-expert | 0.15, 0.25, 0.4, 0.65, 1.0 | 0.6, 0.75, 1.25, 1.6, 2.2 | 45, 60, 90, 150, 300 |
| halfcheetah-medium | 0.005, 0.007, 0.015, 0.02, 0.025 | 0.05, 0.1, 0.2, 0.3, 0.4 | 0.6, 0.75, 1.25, 1.6, 2.2 |
| halfcheetah-medium-replay | 0.005, 0.007, 0.015, 0.02, 0.025 | 1.5, 2, 4.5, 6.5, 8 | 0.6, 0.75, 1.25, 1.6, 2.2 |
| halfcheetah-medium-expert | 0.012, 0.018, 0.024, 0.045, 0.065 | 1.5, 2, 4.5, 6.5, 8 | 45, 60, 90, 150, 300 |

The runner fails closed if a second-round value duplicates a first-round value. Hopper-medium-expert Linear uses a local `15--60` extension rather than a blind `45--300` expansion because its round-one `c=30` choice was accompanied by severe late-stage degradation and non-monotonicity.

## Selection and output semantics

For each dataset and controller family, the runner:

1. audits all four seeds for each of the five new candidates;
2. verifies and loads the SHA-bound parent audit containing the five coarse candidates;
3. rejects any duplicated task/method/parameter identity;
4. ranks all ten candidates by:
   - higher four-seed late-window mean;
   - higher worst-seed late-window mean;
   - smaller mean best-to-late-window drop;
   - smaller numeric parameter;
5. writes combined selections to `TASKWISE_SELECTION.json`.

`TERMINAL_AUDIT.json` separately retains the 540 current-round branches, current-round candidate groups, parent candidate groups, combined ten-point groups, current-round winners, and combined winners.

Task-performance collapse remains unclassified because no threshold is registered. Support/variance boundary events remain unavailable in the unchanged trainer. NaN/Inf numerical failure is audited separately from both.

## Execution

The parent round-one output must remain at:

```text
outputs/e7/d4rl9_glq_taskwise_tuning_run_001/TERMINAL_AUDIT.json
```

and must match SHA-256:

```text
8775edcb436ba759a52eb6b2ae9cdb2cbce966852fd5e8e3739798134523234b
```

Run:

```bash
bash scripts/run_e7_canonical_d4rl9_glq_refinement_one_click.sh
```

The new output root is:

```text
outputs/e7/d4rl9_glq_taskwise_refinement_run_001
```

No run may begin until the implementation SHA is frozen and the proposed experiment is registered through the schema-v3 code-first pilot-registration transaction. Held-out seeds `204--207` remain forbidden in this stage.
