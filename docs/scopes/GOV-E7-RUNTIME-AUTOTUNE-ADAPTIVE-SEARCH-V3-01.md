# GOV-E7-RUNTIME-AUTOTUNE-ADAPTIVE-SEARCH-V3-01 — bounded low-first E7 autotune repair

**Approval:** user explicitly requested immediate repair after the 2026-07-20 E7 P1 launch incidents.  
**Base:** `main@4b718e7439cf78a04f4affa1987ac15582d702d1`.  
**Parent:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`.  
**Scientific impact:** none.  
**Experiment execution:** none in this change.

## Incident

The squared-night E7 adapter inherited the V2 PPO-family candidate grid:

```text
{fallback, 0.50 * safe_cap, 0.75 * safe_cap, safe_cap}
```

and used one `probe_steps` value for both the representative resource probe and every
throughput candidate. On a 198-branch, 200-CPU-pool launch this produced three failures:

1. a large safe ceiling could make the first throughput candidate larger than an already
   observed startup-failure region;
2. a first-candidate failure left no valid candidate and aborted plan instead of retaining a
   conservative lower schedule;
3. setting `probe_steps=100000` and `probe_seconds=2500` multiplied long limits across the
   resource probe and every concurrency candidate, turning preflight into an hour-scale
   sequence of real short trainings.

The existing one-click wrapper also ignored `DRPO_RUNTIME_MAX_WORKERS` and always attempted
`plan` before `run`, even when an immutable selection and run identity already existed.

## Objective

Repair only the squared-night E7 runtime adapter so that:

1. throughput search starts from a guaranteed low candidate and rises monotonically;
2. `--max-workers` remains an optional absolute ceiling, not a prerequisite for finding a
   safe starting point;
3. all squared-night preflight work is capped at 5000 optimizer steps and 300 seconds per
   probe/candidate even when a caller requests larger values;
4. requested and effective probe limits are both preserved in resource identity and evidence;
5. the first failed higher candidate stops upward exploration while preserving already valid
   lower candidates;
6. the one-click wrapper consumes an existing immutable selection through `run --resume`
   instead of re-planning;
7. the unified RunSpec worker ceiling reaches E7 through `DRPO_RUNTIME_MAX_WORKERS` unless the
   E7-specific variable explicitly overrides it.

## Implementation boundary

Modify existing files:

- `src/drpo/e7_squared_exp_night_runtime_autotune.py`;
- `scripts/run_e7_squared_exp_night_one_click.sh`;
- `scripts/run_e7_squared_exp_night_resume_one_click.sh`;
- `tests/test_e7_measured_cpu_wrappers.py`;
- `tests/test_e7_squared_exp_night_runspecs.py`.

Create one documentation file:

- `docs/runtime_resource_autotuning_v3.md`.

The V2 usage document and append-only evolution ledger remain historical authority for V2;
this scoped V3 document records the squared-night successor without rewriting that history.
No new Python file is authorized or required.

## Frozen scientific coordinates

This repair must not change datasets, methods, seeds, controls, coefficients, advantage
estimator, optimizer, batch size, thread environment, training horizon, evaluation cadence,
terminal audit, result status, or experiment responsibility. It changes only engineering
preflight and subprocess-count selection.

## V3 search policy

For the squared-night adapter only:

- selector policy version becomes `3`;
- the candidate sequence is deterministic, sorted, unique, and never above `safe_cap`;
- candidate `1` is always included;
- additional candidates are low-first fractions of the safe ceiling rather than starting at
  50 percent;
- the configured fallback is included only as another bounded candidate;
- requested `probe_steps` and `probe_seconds` are capped to effective values of `5000` and
  `300` respectively for both the representative resource probe and throughput candidates;
- an invalid candidate stops further upward search; already valid lower candidates remain
  eligible under the unchanged retained-peak rule.

## One-click lifecycle

- no selection and no identity: execute `plan`, then `run --resume`;
- both selection and identity present: execute only `run --resume`;
- exactly one of selection or identity present: fail closed as a partial runtime identity;
- `E7_SQUARED_EXP_MAX_WORKERS` overrides `DRPO_RUNTIME_MAX_WORKERS`; otherwise the unified
  value is inherited.

## Explicit exclusions

- dynamic online resizing;
- cross-run cache service;
- reuse of selection across source, binding, or fingerprint changes;
- NUMA or physical-core topology allocation;
- memory cgroup reservation or external-process preemption;
- changing the current immutable selection inside an already planned work directory;
- launching or resuming a scientific run from this development branch.

## Acceptance

1. safe cap 130 produces a low-first grid containing 1 and no candidate above 130;
2. the grid does not require a configured max-worker value to begin below the safe ceiling;
3. requested 100000 steps and 2500 seconds become effective 5000 steps and 300 seconds;
4. every candidate summary records requested and effective limits;
5. adapter installation restores every patched shared-core symbol and policy version;
6. a failed higher candidate leaves completed lower candidates eligible for selection;
7. the E7-specific worker variable overrides the unified variable, while the unified variable
   is inherited when no E7-specific value exists;
8. one-click resumes directly when both immutable identity files exist;
9. one-click fails closed on a partial identity;
10. existing V2 selections are invalid under squared-night V3 and require a new work directory;
11. focused tests, shell syntax, compile, Ruff, full pytest, handoff-authority no-op, and
    governance validators pass on the exact PR head.

## Current-run rule

Existing work directories planned under selector policy version 2 remain immutable historical
artifacts. They may resume only with their original source and policy. They must not be edited
or silently upgraded to V3. A V3 decision requires a new work directory and new selection.
