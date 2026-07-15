# EXT-H-E7-SQUARED-EXP-KL-TUNE-01

## Document status

- type: experiment proposal only;
- registration status: **not registered**;
- implementation status: **not implemented**;
- execution authorization: **none**;
- predecessor: `EXT-H-E7-SQUARED-EXP-NIGHT-01`;
- proposed scientific class: Hopper/Walker external-validity development screening;
- proposed result-status ceiling: `pilot`;
- proposal base: `main` commit `db663872564547f73a20b633bd231f76785a2a2d`.

This file is not a second research master and does not authorize a launch. Before implementation or execution, the selected matrix must be registered through the schema-v3 handoff/registry authority path.

## 1. Evidence motivating the successor

The completed predecessor used

\[
w(d)=w(0)\exp\left[-c(d/2)^2\right]
\]

with `w(0)=1`, `c in {0.25,0.5,1,2,4,8}`, fixed-K4 PPO, and a `K_max=16` PPO path that refreshed the old policy whenever analytic diagonal-Gaussian `KL(old || new)` exceeded `0.01`.

The compact predecessor evidence records:

- 126/126 branches completed at 1M updates;
- terminal audit `PASS` and zero NaN/Inf failures;
- KL-refresh beat fixed K4 in `16/21` paired cells;
- mean 800k--1M difference `+3.79`;
- at `c in {2,4,8}`, mean difference `+5.52`, with `7/9` wins;
- the mean realized old-policy reference lifespan was about `3.93` updates;
- mean two-seed late-window SD was `7.03` for KL-refresh and `5.87` for K4.

Therefore the locked interpretation is:

1. adaptive KL-triggered refresh has a positive finite-horizon mean-performance signal;
2. the evidence does not show lower seed variance;
3. the evidence does not establish that fixed longer reuse is beneficial;
4. the useful `c` region may extend beyond `8`, but this has not been tested;
5. GAE has no result because zero GAE branches started.

Evidence paths:

- `experiments/results/e7_squared_exp_night_1m_pilot/RESULT_SUMMARY.json`;
- `experiments/results/e7_squared_exp_night_1m_pilot/actor_comparisons.csv`;
- `experiments/results/e7_squared_exp_night_1m_pilot/terminal_audit.json`.

## 2. Claims to test

### Q1: KL refresh threshold

Determine whether the positive KL-refresh signal is robust to the refresh threshold, and whether one common threshold provides a useful trade-off between stale-reference reuse and excessively frequent refresh.

### Q2: high-`c` squared-EXP region

Determine whether the common useful region continues beyond `c=8`, or whether very large `c` merely approaches Positive-only behavior without preserving useful near-negative information.

### Q3: generic actor-lifecycle effect versus negative-control interaction

Determine whether KL refresh helps Positive-only and squared-EXP branches similarly, or whether its benefit is specifically larger when signed negative updates are present.

## 3. Explicit exclusions

The proposed sweep must not:

- add GAE branches;
- use held-out seeds `204--207` during tuning;
- change the canonical actor or critic architecture;
- change critic target, expectile loss, one-step TD advantage, actor-before-critic order, optimizer, batch size, learning rate, PPO clip epsilon, dataset versions, horizon, or evaluation protocol;
- introduce a KL penalty, entropy bonus, actor-gradient clipping, or value clipping;
- compare new taper families;
- claim convergence, steady state, universal PPO superiority, controlled causal identification, OOD generalization, or a formal D4RL method ranking;
- select different `c` values per dataset and present them as one common method.

GAE must remain a separate successor because it changes the advantage estimator and currently lacks a verified ordered-trajectory and terminal/truncation contract.

## 4. Shared frozen settings

Unless the proposal is revised and explicitly approved before registration, both tuning stages use:

- datasets:
  - `hopper-medium-expert-v2`;
  - `walker2d-medium-v2`;
  - `walker2d-medium-replay-v2`;
- kernel: `w(d)=w(0) exp[-c(d/2)^2]`;
- `w(0)=1` for squared-EXP and `w(0)=0` for Positive-only;
- reference distance: `2.0`;
- horizon: `1,000,000` optimizer updates;
- 500k: intermediate checkpoint only;
- terminal window: `800,000--1,000,000`;
- evaluation: every `50,000` updates, ten episodes;
- PPO clip epsilon: `0.2`;
- batch size: `256`;
- learning rate: `3e-4`;
- one-step TD advantage;
- paired branch construction and diagnostics inherited from the predecessor.

## 5. Stage A: KL-threshold and reference-lifecycle screen

### 5.1 Development seeds

Use existing tuning seeds:

```text
200, 201
```

These seeds are already development evidence and may be reused for threshold screening. They must not be presented as fresh confirmation.

### 5.2 Weight controls

```text
Positive-only
c = 4
c = 8
c = 16
c = 32
```

`c=4` and `c=8` anchor the predecessor. `c=16` and `c=32` probe the requested region above `12` without filling a dense grid before the KL lifecycle is calibrated.

### 5.3 Actor-reference lifecycle controls

For every weight control, compare:

1. fixed K4 PPO;
2. fixed K16 PPO with no KL-triggered early refresh;
3. adaptive `K_max=16`, `target_kl=0.003`;
4. adaptive `K_max=16`, `target_kl=0.01`;
5. adaptive `K_max=16`, `target_kl=0.03`.

The fixed-K16 branch is required. Without it, a KL-threshold sweep cannot separate the effect of adaptive refresh from the effect of allowing a longer maximum reference window.

### 5.4 Branch count

\[
5\text{ controls}\times5\text{ lifecycles}\times3\text{ datasets}\times2\text{ seeds}=150\text{ branches}.
\]

All branches run to 1M unless a NaN/Inf numerical failure occurs.

### 5.5 Stage-A qualification rule

A KL threshold is eligible for Stage B only when all of the following hold:

1. all expected branches are present and terminal-audited;
2. no NaN/Inf numerical failure is hidden or merged with task degradation;
3. its pooled paired 800k--1M mean difference versus fixed K4 is positive;
4. its pooled paired median difference versus fixed K4 is positive;
5. it wins more than half of the paired `(dataset, control)` cells versus fixed K4;
6. its result is reported separately for each dataset and for Positive-only versus squared-EXP controls.

If multiple thresholds qualify, select the one with the largest pooled paired late-window mean difference. Ties are broken by median difference, then by paired-cell win count. Seed SD is reported but is not silently converted into a selection objective.

If no threshold qualifies, Stage B does not start. The correct result is “no common KL threshold qualified in Stage A,” not a post hoc dataset-specific selection.

## 6. Stage B: high-`c` refinement on fresh development seeds

### 6.1 Development confirmation seeds

Use:

```text
202, 203
```

These remain development seeds, not held-out confirmation seeds. They provide a fresh check relative to the `200,201` threshold screen. Held-out seeds `204--207` remain untouched.

### 6.2 Weight controls

```text
Positive-only
c = 1
c = 2
c = 4
c = 8
c = 12
c = 16
c = 24
c = 32
```

The grid preserves lower anchors while resolving the transition from the predecessor's upper boundary `8` into the requested `>12` region.

### 6.3 Actor modes

Compare only:

1. fixed K4 PPO;
2. the single Stage-A-selected adaptive KL lifecycle.

No A2C, fixed K16, or additional KL threshold enters Stage B. This prevents the high-`c` refinement from becoming a second uncontrolled factorial expansion.

### 6.4 Branch count

\[
9\text{ controls}\times2\text{ lifecycles}\times3\text{ datasets}\times2\text{ seeds}=108\text{ branches}.
\]

### 6.5 Stage-B outputs

For every `(dataset, lifecycle, c)` group, report:

- 500k intermediate return;
- BEST and best step;
- FINAL;
- BEST-to-FINAL drop;
- 800k--1M mean, standard deviation, and slope;
- paired difference versus Positive-only under the same lifecycle;
- paired difference between adaptive KL and fixed K4 at the same `c`;
- analytic KL, trigger fraction, reference-block lifespan, ratio-outside fraction, and true objective-clip fraction;
- effective negative-weight quantiles and threshold fractions;
- task-performance degradation/collapse, support or variance-boundary event, and NaN/Inf numerical failure as separate fields.

### 6.6 Stage-B interpretation rule

Stage B produces a **common shortlist**, not an automatic winner.

A `c` value may enter the common shortlist only when:

1. all six `(dataset, seed)` branches for the selected adaptive lifecycle complete;
2. its pooled paired late mean versus Positive-only is positive;
3. it is not negative versus Positive-only on all seeds of any one dataset;
4. its pooled adaptive-KL versus fixed-K4 difference is non-negative;
5. no numerical failure occurs;
6. the result is not created by choosing a different `c` for each dataset.

If no `c` satisfies these conditions, report that no common high-`c` setting qualified. Dataset-specific optima may be shown descriptively but may not be promoted to one method configuration.

## 7. Total immediate tuning matrix

```text
Stage A: 150 branches
Stage B: 108 branches
Total:   258 branches
```

The predecessor's 126-branch run took about 2.69 wall-clock hours at the reported 126-worker setting. A rough planning estimate for 258 comparable branches is approximately twice that compute volume, but runtime must be re-measured by liveness and resource probing rather than treated as a guarantee.

## 8. Held-out confirmation gate

Held-out seeds `204--207` are not part of this tuning proposal's immediate matrix.

After Stage B, a separate registered confirmation stage may compare only:

- Positive-only;
- fixed K4 at the frozen common `c`;
- adaptive KL at the same frozen common `c` and threshold.

That confirmation would contain:

\[
3\text{ methods}\times3\text{ datasets}\times4\text{ held-out seeds}=36\text{ branches}.
\]

It requires a new explicit user approval and registry/handoff freeze. Tuning results must not access these seeds first.

## 9. Separate GAE successor

Proposed future ID:

```text
EXT-H-E7-PPO-GAE-01
```

This future experiment remains blocked until the code provides and tests:

1. stable ordered trajectory or episode identity;
2. terminal versus time-limit truncation semantics;
3. exact discount and bootstrap semantics;
4. separation of training trajectories from evaluation evidence.

The first GAE experiment should be minimal: one-step TD versus GAE `lambda=0.95`, using only Positive-only and one frozen squared-EXP `c`, under fixed K4 and the frozen adaptive-KL lifecycle. It should not begin with a wide lambda grid.

## 10. Required implementation work for the next session

1. create a successor runner and config rather than mutating the frozen predecessor;
2. parameterize `target_kl` in branch identity, branch config, bootstrap, diagnostics, and aggregation;
3. add a fixed-K16/no-KL control;
4. keep the exact `variant=iqlv_exp_rank` trainer plumbing;
5. add an end-to-end command-contract test that expands a branch command, reaches canonical argparse, and performs a bounded update smoke;
6. enforce the Stage-A-to-Stage-B qualification rule in the launcher or require an immutable selected-threshold record before Stage B;
7. keep `204--207` forbidden in both tuning stages;
8. create terminal audit and compact result deposition paths before launch;
9. register the experiment through one schema-v3 delta before any formal or held-out execution;
10. preserve all predecessor files and results.

## 11. Required execution sequence

1. review and approve this proposal;
2. register the frozen successor matrix;
3. implement the minimal science path and command-contract tests;
4. run real-data liveness;
5. start Stage A;
6. terminal-audit Stage A and materialize the threshold-selection record;
7. start Stage B only if a threshold qualifies;
8. terminal-audit and package Stage B;
9. decide whether to register held-out confirmation;
10. develop the GAE trajectory contract in a separate task.

## 12. Remaining uncertainties

- the compact repository evidence does not include all predecessor raw trajectories;
- the best KL threshold may depend on dataset or negative-control strength;
- very large `c` may become functionally close to Positive-only;
- two development seeds per stage are still screening evidence;
- no GAE implementation or result exists yet;
- fixed 1M training does not establish convergence or steady state.
