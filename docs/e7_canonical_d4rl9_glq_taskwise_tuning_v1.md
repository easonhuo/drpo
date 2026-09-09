# E7 D4RL-9 task-wise Global / Linear / Quadratic tuning pilot

## Scope

This is a non-destructive pilot substage of `EXT-H-E7-BENCH-01`. It reuses the canonical joint actor-critic trainer and runtime injection path. It tunes only three negative-gradient controllers over all nine D4RL locomotion cells:

- Global negative scaling;
- Reciprocal-Linear distance taper;
- Reciprocal-Quadratic distance taper.

Exponential is not searched in this stage. Positive-only and Exponential are also not rerun as training branches, because this stage only selects task-specific hyperparameters. A later held-out confirmation must restore the required anchors before any method ranking or paper-table claim.

This pilot is external-validity parameter selection. It does not replace the controlled C-U1 or D-U1 mechanism environments and cannot establish convergence or steady-state ranking.

## Frozen training protocol

The server-local canonical contract and nine-task run spec remain the source of dataset paths and SHA-256 identities. The runner requires exactly these datasets, in this order:

1. `hopper-medium-v2`
2. `hopper-medium-replay-v2`
3. `hopper-medium-expert-v2`
4. `walker2d-medium-v2`
5. `walker2d-medium-replay-v2`
6. `walker2d-medium-expert-v2`
7. `halfcheetah-medium-v2`
8. `halfcheetah-medium-replay-v2`
9. `halfcheetah-medium-expert-v2`

The unchanged trainer must expose `variant=iqlv_exp_rank`, `alpha=0.11`, `tau=0.5`, `temp=5.0`, batch size `256`, learning rate `3e-4`, one million updates, evaluation every `50k` updates, and ten rollout episodes per evaluation.

Development seeds are `200,201,202,203`. Held-out seeds `204,205,206,207` are forbidden in this stage.

## Controller grids

The standardized radius is

\[
r=\sqrt{\frac{1}{d}\sum_j\left(\frac{a_j-\mu_j}{\sigma_j}\right)^2},
\qquad u=r/R_{\mathrm{ref}},
\qquad R_{\mathrm{ref}}=2.
\]

Positive advantages are unchanged. Only negative advantages are multiplied by the controller-specific factor.

### Global

\[
w_{\mathrm G}=\alpha s,
\qquad
s\in\{0.001,0.003,0.01,0.03,0.1\}.
\]

### Reciprocal-Linear

\[
w_{\mathrm L}(r)=\alpha\frac{1}{1+c u},
\qquad
c\in\{0.5,1,3,10,30\}.
\]

### Reciprocal-Quadratic

\[
w_{\mathrm Q}(r)=\alpha\frac{1}{1+c u^2},
\qquad
c\in\{0.5,1,3,10,30\}.
\]

The reciprocal families fix `negative_scale=1`. `R_ref` is fixed and cannot be co-tuned with `c`. Each family receives exactly five candidates, yielding `9 datasets × 4 seeds × 15 candidates = 540` branches.

## Selection and terminal audit

Selection is independent for each dataset and method. The primary metric is the four-seed mean of the late-window score over evaluations at `750k, 800k, 850k, 900k, 950k, 1M` updates.

The deterministic tie-break order is:

1. higher four-seed late-window mean;
2. higher worst-seed late-window mean;
3. smaller mean best-to-late-window drop;
4. smaller numeric hyperparameter.

The audit also reports final score, best score and step, best-to-final drop, terminal slope, and finite-score checks. Task-performance collapse remains unclassified because no threshold is registered. Support/variance boundary events are unavailable in the unchanged trainer. NaN/Inf numerical failure is reported separately from those two event classes.

The output `TASKWISE_SELECTION.json` is a development selection artifact only. It cannot populate the formal D4RL-9 table and cannot establish a cross-method ranking. Confirmation on untouched seeds is required.

## One-click execution

The server paths intentionally match the prior canonical nine-task runs:

```bash
bash scripts/run_e7_canonical_scale1_grid_one_click.sh
```

The script expects:

- `/root/d4rl2/configs/e7_canonical_contract_9task.json`
- `/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json`

It writes to `outputs/e7/d4rl9_glq_taskwise_tuning_run_001`, supports identity-checked resume, and produces:

- `EXECUTION_PLAN.json`
- `RUN_IDENTITY.json`
- `RUN_SUMMARY.json`
- `TERMINAL_AUDIT.json`
- `TASKWISE_SELECTION.json`
