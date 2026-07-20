# PAPER-CODE-VALIDATION Hopper config-path defect 01

## Identity

- Claim: `PAPER-CODE-VALIDATION-01`
- Parent claim: `PAPER-CODE-REFERENCE-01`
- Defect ID: `PAPER-CODE-VALIDATION-HOPPER-CONFIG-PATH-01`
- Scientific status impact: none
- Numerical or functional behavior impact: none

## Old statement

The task-local acceptance, migration, and validation matrices listed
`configs/e7_hopper_q2.yaml` as an authoritative Hopper E7-Q2 source.

That path does not exist on either the development branch or current `main`.

## Repository evidence

The actual public launcher `scripts/run_e7_hopper_q2.py` defines:

```text
DEFAULT_CONFIG = "configs/e7_hopper_q2_medium_replay_v2.yaml"
```

The launcher validates that this configuration exists inside the repository,
is tracked by Git, and is included in the hardened source-file manifest before
it delegates to `src/drpo/e7_hopper_q2.py`.

At the integration-freshness audit boundary, the actual configuration has the
same Git blob on both refs:

- current `main`: `02e6e463256dcde96d640a42b6fbf8534fac8774`;
- `dev/paper-code-reference-01`: `02e6e463256dcde96d640a42b6fbf8534fac8774`.

The Hopper scientific runner and launcher are also identical across the two
refs at this boundary:

- `src/drpo/e7_hopper_q2.py`: `a93eda3e94429a8764de9d6c35a98e09e6a0a14d`;
- `scripts/run_e7_hopper_q2.py`: `d3d964f3576f8c66a911c6b9f3cb1023e1985287`.

## Repair

Replace only the stale task-local document reference:

```text
configs/e7_hopper_q2.yaml
```

with:

```text
configs/e7_hopper_q2_medium_replay_v2.yaml
```

in the live paper-code validation documents that contain the old path.

## Boundaries

This repair:

- does not change Hopper code, configuration content, dataset identity, seeds,
  budgets, thresholds, critic lifecycle, advantages, methods, or rollout rules;
- does not launch a smoke, pilot, or formal run;
- does not alter any scientific result or authorize a method ranking;
- does not claim that Hopper HDF5/MuJoCo liveness has passed;
- preserves the distinction between Hopper E7-Q2 mechanism validation and
  D4RL-9 task-performance validation.
