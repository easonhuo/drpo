# Development scope: E8 multitask AsymRE capability

## Status

Code-only development scope. No scientific experiment is launched by this change, and no new formal experiment ID, final five-point `delta_v` grid, RunSpec, method ranking, or result claim is frozen here.

Base repository state for this task: `easonhuo/drpo@8a21fd16e63ea96a0cde2a477353d24562624655` on `main`.

## Approved responsibility

Extend the existing E8 multitask cold-start orchestration so a successor config can select the already implemented canonical AsymRE family and sweep `delta_v` across the existing eight P0 transfer tasks.

The scientific AsymRE objective is not reimplemented in `src/drpo/e8_multitask_exp_tuning.py`. The existing canonical paper-aligned implementation remains authoritative: `A = R - delta_v`, equivalently branch coefficients `1-delta_v` for the positive branch and `1+delta_v` for the negative branch under signed rewards `+1/-1`.

The multitask layer may only add method/config/cell representation, planning, dispatch, manifest/provenance plumbing, and regression tests required to route AsymRE cells through the existing canonical trainer and task adapters.

## Allowed implementation files

- `src/drpo/e8_multitask_exp_tuning.py`
- `src/drpo/e8_experiment_config.py`
- extensions to existing `tests/test_e8_multitask_p0.py`
- this development scope record

No new Python path is approved or required.

## Required design properties

1. Historical EXP cold-start configs and their immutable identities must continue to validate unchanged.
2. The existing canonical scientific kernel remains import-only; no AsymRE loss formula may be copied into the multitask orchestration.
3. A successor AsymRE config must be able to define method, task-local or shared `delta_v` points, seed offsets, and expected cell count without adding experiment-specific parameter tuples to Python.
4. AsymRE dispatch must construct the existing canonical paper cell with `family="asymre"`, `alpha=1+delta_v`, and zero taper coefficient, so the existing trainer applies the signed objective coefficients.
5. AsymRE uses no distance/remoteness taper and no value network.
6. Plan/cell identity must record method and `delta_v` explicitly enough to prevent collision with EXP cells and to support exact resume/provenance checks.
7. Existing task data, bank selection, verifier/evaluator adapters, optimizer horizon, evaluation semantics, numerical-failure reporting, and terminal audit responsibilities are not changed by this capability.

## Explicit exclusions

This change must not:

- choose or freeze the final five `delta_v` values for the baseline comparison;
- choose the final baseline-comparison experiment ID or claim wording;
- add TOPR or DPO capability;
- alter frozen seeds, training horizon, data scale, negative-bank construction, task runtime overrides, or evaluation thresholds;
- change the canonical Countdown AsymRE implementation;
- start a smoke, pilot, or formal scientific run;
- claim any AsymRE performance result or method ranking;
- modify `docs/handoff.md` or `experiments/registry.yaml`.

## Acceptance criteria for this development task

- Existing historical cold-start config validation remains unchanged.
- Regression tests show that a synthetic successor AsymRE config expands the requested config-defined cells and seed offsets without hard-coded grid constants.
- Regression tests show that AsymRE cell identity/key/plan exposes `delta_v` and does not collide with EXP identity.
- Regression/static inspection shows canonical dispatch uses `family="asymre"` and does not add an AsymRE loss implementation to `e8_multitask_exp_tuning.py`.
- Applicable existing unit tests pass before the Draft PR is presented for review.
