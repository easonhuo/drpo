# EXT-H-E7-Q2 Hopper Q2 one-command run

This runner is the external Hopper learned-critic mechanism validation. It does
not replace the controlled C-U1 causal experiment and does not authorize a
method-ranking claim.

## One command

Place `hopper_medium_replay-v2.hdf5` in one of the registered locations, or set:

```bash
export DRPO_HOPPER_MEDIUM_REPLAY=/absolute/path/hopper_medium_replay-v2.hdf5
```

Then run the formal protocol from a clean checkout of current `main`:

```bash
python3 scripts/run_e7_hopper_q2.py
```

The launcher defaults to `--run-class formal`, creates a timestamped directory
under `runs/e7_q2/`, invokes the hardened experiment guard, writes progress and
failure evidence persistently, and creates the final result ZIP next to the run
directory. No interactive decision is required.

Useful variants:

```bash
# Resolve paths and show the exact command without training.
python3 scripts/run_e7_hopper_q2.py --plan-only

# Explicit dataset and output root.
python3 scripts/run_e7_hopper_q2.py \
  --dataset-path /root/d4rl/d4rl_datasets/locomotion/hopper_medium_replay-v2.hdf5 \
  --output-root /root/e7_runs

# Small diagnostic only; never formal evidence.
python3 scripts/run_e7_hopper_q2.py --run-class pilot --allow-dirty
```

## Protocol v4.3: fixed-budget long-run

The formal run no longer lets a short-window stationarity gate decide when
training stops. Fixed budgets provide equal horizons and reproducibility;
terminal audits remain mandatory but are post-hoc classifications rather than
stopping rules.

### Canonical critic

- Train for exactly **100,000 optimizer steps**.
- Evaluate every **2,000 steps**.
- Stop early only for a non-finite loss, gradient, parameter, or unrecoverable
  process failure.
- Select the checkpoint with the lowest validation MSE over the full budget.
- Preserve the final checkpoint as a comparator.
- Report validation R²/Pearson, best/final prediction ratios, advantage sign and
  rank stability, and negative-set overlap as quality diagnostics.
- Formal operational acceptance requires the fixed budget to complete and the
  selected metrics to be finite. Fixed-budget completion is not described as
  optimizer convergence.

The previous stationarity candidate, relative-update, slope, and exact-2x
extension calculations are retained for diagnosis and historical comparability,
but they do not stop critic training or choose the canonical checkpoint.

### Actor stages

- Positive-only initialization: exactly **100,000 optimizer steps**.
- Each downstream branch: exactly **200,000 optimizer steps** from the same
  Positive-only checkpoint.
- Actor audit interval: **5,000 steps**.
- Rollout interval: **25,000 steps** with **5 episodes** at intermediate milestones;
  the fixed-budget final checkpoint is re-evaluated with **20 paired episodes**.
- Stop early only for NaN/Inf or another explicit numerical failure.

All five negative-update branches therefore have the same horizon:

- `signed`: the full signed-advantage baseline, retaining both positive and
  negative advantages without a near/far intervention;
- `near_zero`;
- `far_zero`;
- `far_cap`;
- `dynamic_budget_matched_global`.

The post-hoc terminal audit separately reports finite terminal behavior,
persistent or slow drift, fixed-horizon inconclusive behavior, task-performance
collapse, support/variance-boundary events, and NaN/Inf numerical collapse. A
fixed horizon by itself never establishes convergence. A finite-terminal label
still requires a candidate state, exact 2x continuation, and no support-boundary
event.

## Mechanism and controls

The core per-seed mechanism gate contains only:

- natural far-field presence;
- corrected Gaussian log-scale quadratic geometry and analytic/autograd agreement;
- measurable far/near full-parameter gradient amplification.

Whether the log-scale branch dominates the mean branch is reported as a
diagnostic, not treated as a necessary mechanism condition.

Control outcomes are not collapsed into one permissive boolean. Each control
reports diagnostic-score mitigation, support-boundary rescue, task-performance
rescue, and finite-terminal rescue separately.

`dynamic_budget_matched_global` recomputes a detached global scale for every
batch so that its negative influence proxy
`sum(|A| * joint_output_score)` matches what the registered Far-cap would retain
on that same batch. This is a proxy budget match, not a claim of exact
full-parameter-gradient equality.

## Artifact compatibility

Canonical critic artifacts use schema version **3** and include the runner
version, config hash, run class, dataset identity, loaded transition count,
dimensions, canonical critic seed, and fixed-budget protocol identity. Schema-2,
pilot, old v4.1/v4.2, or otherwise mismatched artifacts fail closed rather than
being silently reused.

## Validation/test separation

Checkpoint selection uses validation MSE only. Test R²/Pearson remain
final-report-only and never select or admit a checkpoint. Critic quality and
advantage-stability diagnostics must be reported even though they no longer
control the fixed training budget.
