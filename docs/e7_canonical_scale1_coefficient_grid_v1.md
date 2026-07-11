# E7 canonical scale-1 coefficient tuning pilot

## Scope

This is a parameter-tuning substage of `EXT-H-E7-BENCH-01`. It uses the existing canonical joint actor-critic path and does not introduce a new algorithm or a formal D4RL-9 protocol.

The previous pilot multiplied every distance-tapered negative advantage by a global `negative_scale`. The strongest unified FINAL result used Exponential with `negative_scale=0.1`. That setting also reduced near-field negative advantages by a factor of ten. This substage tests the narrower question: can the scale be fixed to one, preserving the full canonical negative coefficient at zero distance, while stronger distance decay controls the far field?

## Locked variables

The canonical alpha remains `0.11`, reference distance remains `2.0`, and all taper branches use `negative_scale=1.0`. Dataset identities, seeds, one-million-step horizon, actor-critic implementation, optimizer, batching, evaluation cadence, and result provenance must come from the unchanged run spec and canonical contract.

Only one parameter changes within each taper family:

| Family | Coefficient grid |
|---|---|
| Reciprocal-linear | `0.4362580032734791, 1, 3, 10, 30` |
| Reciprocal-quadratic | `0.5520268617673281, 1, 3, 10, 30` |
| Exponential | `0.374162511054291, 0.75, 1.5, 3, 6` |

Positive-only and canonical signed remain controls. The run spec may additionally provide the unchanged original ExpRank-MR passthrough.

## Selection and reporting

The primary tuning metric is the two-seed mean FINAL score at one million updates, aggregated over the same nine tasks used by the preceding pilot. Best-in-trajectory, late-window mean, best-to-final drop, task-performance collapse, support or variance-boundary events, and NaN/Inf numerical failure remain separately reported diagnostics.

This run is pilot-only. It may select a candidate coefficient for a later frozen evaluation, but it cannot populate a formal nine-task table or establish a formal method ranking.

## Entry point

Use the existing generic launcher with:

```bash
python scripts/run_e7_canonical_scale1_grid.py plan \
  --contract /absolute/path/to/canonical_contract.json \
  --run-spec /absolute/path/to/the_existing_9task_run_spec.json \
  --grid configs/e7_canonical_scale1_coefficient_grid_v1.json \
  --work-dir /absolute/path/to/new_plan_dir \
  --max-workers 120
```

Replace `plan` with `run` only after plan review and the normal liveness gate. No Hopper training is part of this repository update.
