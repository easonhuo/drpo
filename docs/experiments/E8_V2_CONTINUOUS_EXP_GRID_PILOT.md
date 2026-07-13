# E8 V2 continuous EXP grid pilot

## Status

- experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01`;
- class: Countdown external-validity development tuning pilot;
- implementation route: code-first dev branch at explicit user request;
- authoritative handoff/registry registration: pending;
- result: `not_run`;
- formal ranking, convergence, steady state, and OOD claims: forbidden.

The predecessor 72-cell sweep is preserved as historical evidence for an extreme-selection and initialization-gradient-budget-matched design. It is not overwritten and does not answer the continuous fixed-alpha question below.

## Training definition

For prompt `p`, let `K_p` be the number of unique negative expressions after applying the existing Countdown expression cleaner and retaining the first occurrence. Every unique negative participates in training. For negative `i`,

\[
d_{p,i}=-\operatorname{stopgrad}(\ell^-_{p,i}),\qquad
u_{p,i}=d_{p,i}/2,
\]

and

\[
w_{p,i}=\alpha\exp(-c u_{p,i}^{2}).
\]

The loss is

\[
L_p=-\ell^+_p+\frac{1}{K_p}\sum_{i=1}^{K_p}w_{p,i}\ell^-_{p,i}.
\]

The denominator is `K_p`, never the sum of weights. No scale is introduced to restore the negative-gradient budget removed by tapering.

Special cases are exact:

- `alpha=0`: Positive-only;
- `alpha>0, c=0`: Global with fixed coefficient `alpha` on every unique negative;
- `alpha>0, c>0`: continuous EXP taper.

## Frozen development grid

- `alpha`: `0, 0.025, 0.05, 0.11, 0.25, 0.5, 1.0`;
- `c`: `0, 0.25, 0.5, 1.0, 1.5`;
- `alpha=0` is deduplicated across `c`;
- unique parameter points: `31`;
- fresh development seed offsets: `3000, 4000`;
- total cells: `62`;
- fixed horizon: `1200` optimizer steps;
- no early stop;
- validation Greedy and Pass@8 every `100` steps;
- validation Pass@64 every `200` steps;
- tuning does not read the test split.

The seed offsets are development-only coordinates selected after the predecessor used offsets `0, 1000, 2000`. They require authoritative review before any later confirmatory registration.

## Forbidden implementation behavior

The training path must not call current-bank argmin/argmax selection, create near/far training classes, use a 0.5/0.5 binary mixture, calibrate `negative_scale`, match gradient RMS, normalize by the weight sum, or activate SBRC, Hybrid, entropy bonuses, dynamic alpha, SFT initialization, on-policy sampling, or replay refresh.

Near/far language may appear only in post-hoc distance-quantile diagnostics and is not part of the loss.

## Execution

The opt-in dev launcher performs:

1. strict config and input validation;
2. GPU visibility, utilization, free-VRAM, and host-memory slot selection;
3. one actual two-step representative liveness run on the selected hardware;
4. only after liveness passes, the resumable 62-cell sweep with one process per selected GPU;
5. terminal aggregation and separate task/validity/numerical reporting.

The liveness output and any dev-branch run are engineering or tuning evidence only. They cannot be promoted automatically into formal evidence.
