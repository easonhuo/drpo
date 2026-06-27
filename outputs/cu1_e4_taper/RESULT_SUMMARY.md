# C-U1-E4-TAPER-01 Formal Result Report

## Scope and provenance

- Experiment: `C-U1-E4-TAPER-01`
- Base commit: `054c2e275cfd36e07e8883cb65d0b8df00460361`
- Source mode: exact complete Git bundle; local `HEAD` and bundle `origin/main` matched the same full SHA.
- Worktree at launch: clean.
- Formal seeds: 70--89.
- Matrix: 20 paired seeds x 11 registered method/configuration conditions = 220 runs.
- Environment: C-U1, same-distribution held-out-context generalization. This is not an OOD protocol.
- The first launch attempt was terminated by the execution-tool foreground time limit (`SIGTERM`, return code `-15`) during seed 70. Its verified failed-run artifact was preserved and was not reused as scientific evidence. The successful attempt used a fresh output root (`run_002`).

## Execution outcome

The successful guarded attempt completed all 220 registered runs with process exit code 0. Raw computation, generic integrity checks, aggregation, paired inference, and terminal classification all completed. Runtime was approximately 71 minutes on CPU.

The scientific status is **finite-step validated / 有限训练步数验证**, not long-run validated. The terminal audit did not pass because 200 controlled/positive runs reached the frozen 8,000-step maximum without a qualifying stable candidate and therefore without a 2x continuation. The 20 unweighted runs terminated at a registered support/variance boundary event.

## Registered primary comparison: quadratic vs linear at rho=0.25

At matched reference attenuation and paired seeds:

- reciprocal-quadratic had a lower terminal full-parameter far/near negative-gradient ratio in **20/20** seeds;
- reciprocal-quadratic had a higher held-out-context reward in **20/20** seeds;
- mean reward difference (quadratic minus linear): **+0.011372**, paired bootstrap 95% interval **[+0.010951, +0.011826]**;
- mean far/near-ratio difference (quadratic minus linear): **-1.601377**, paired bootstrap 95% interval **[-1.617527, -1.586817]**;
- aggregate terminal far/near ratio: linear **2.274700**, quadratic **0.673323**;
- aggregate held-out-context reward: linear **0.633335**, quadratic **0.644707**.

This finite-horizon result supports the preregistered mechanism-order claim that reciprocal-quadratic suppresses far-field negative gradients more strongly than reciprocal-linear under the shared standardized distance. Because neither family passed the frozen terminal-state audit, it does not establish a terminally stable method ranking.

## Descriptive secondary observations

At rho=0.25, exponential tapering had aggregate reward **0.650534** and terminal far/near ratio **0.295489**. Positive-only had aggregate reward **0.646791**. These are descriptive outcomes only: the experiment did not preregister a universal winner, the methods were not negative-gradient-budget matched, and the controlled runs did not reach a 2x-confirmed terminal plateau.

The same descriptive ordering of far-field suppression persisted at rho=0.50 and rho=0.75: reciprocal-quadratic ratios were **0.733672** and **0.769890**, versus reciprocal-linear **2.392143** and **2.458452**.

## Failure and boundary events reported separately

Across all 220 runs:

- task-performance collapse events: **10**;
- support/variance-boundary events: **20**;
- NaN/Inf numerical events: **0**.

All 20 unweighted runs hit the support/variance boundary at step 100; 10 of those also met the task-performance-collapse criterion. No run produced a NaN/Inf numerical failure. Controlled taper and positive-only runs had no registered task-collapse or support-boundary event within 8,000 steps, but all remained terminally unresolved under the frozen stationarity and 2x rules.

## Interpretation boundary and next gate

Allowed conclusion: at the frozen 8,000-step horizon, reciprocal-quadratic gives substantially stronger far-field suppression than reciprocal-linear at matched reference attenuation, with a paired reward advantage in this C-U1 same-distribution held-out-context experiment.

Not allowed: long-run/stable-fixed-point validation, OOD generalization, universal method ranking, or the claim that exponential is generally best.

No automatic extension, threshold relaxation, optimizer change, or method redefinition is authorized. The next project step is the already registered E6 review/freeze workflow; any new E4-TAPER convergence-resolution experiment requires a separate document-first registration.

## Environment-identification and comparison-fairness note (v44)

The equal negative reward/advantage values are a controlled construction, not an accidental property of natural behavior-policy sampling. C-U1 remains a continuous-action, continuous-reward environment; the formal offline dataset selects eight points from a continuous equal-reward contour per state. This isolates quality magnitude from policy-relative distance inside the negative set.

Directional utility is not claimed to be independent of distance. In the registered 2D geometry, local negative repulsion can align with the hidden-optimum direction, while far-side repulsion can become misaligned. The experiment therefore studies a controlled informativeness-amplification mismatch. It does not establish a universal near-negative-good/far-negative-bad law.

Linear, Quadratic, and Exponential were matched at the reference point `w(d_ref)=rho`, but were not matched in slope, average near retention, total negative-gradient norm, or cumulative optimizer update. The formal result is an anchor-normalized mechanism-order result. A best-tuned family ranking, selective Distance-versus-Global superiority, and stable terminal ranking require separately registered near-retention/budget matching, independent confirmatory seeds, and long-run/2x audits.

The analytic `p=2` boundary concerns asymptotic boundedness of the learnable Gaussian log-scale output-gradient branch under the registered assumptions. It does not by itself imply that Quadratic must have higher task reward than every tuned Linear control, nor that Exponential must be the best task method.
