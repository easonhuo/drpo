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

## Protocol v4.2 acceptance pipeline

The critic pipeline now separates three questions that were previously mixed:

1. **Optimizer stationarity.** A fixed training-audit loss, validation-MSE slope,
   relative parameter movement, and exact 2x continuation are reported. Raw
   whole-network gradients and updates remain diagnostics only.
2. **Checkpoint selection.** A genuinely terminal extension checkpoint is selected only when it passes stationarity and remains within the registered final/best validation-MSE ratio; otherwise the lowest validation-MSE checkpoint is selected.
3. **Frozen-advantage acceptance.** Formal use requires held-out predictive
   quality and stability between the selected and final continuation checkpoints:
   advantage sign agreement, Pearson/Spearman correlation, and negative-set
   Jaccard overlap. `optimization_terminal` is never forced to `true`.

The actor terminal audit likewise uses relative parameter movement rather than a
model-size-dependent raw update norm. Candidate terminality uses scale-normalized
window drift of the registered policy states (`mean_abs`, `sigma_mean`, and
`phantom_distance_mean`) and then requires an exact 2x continuation. The
positive NLL slope remains diagnostic because it can cross zero and fluctuate
under minibatch evaluation even when the policy state is bounded. Task-performance
collapse, support/variance-boundary events, NaN/Inf numerical collapse,
persistent drift, and finite terminal states remain separate outputs.

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
on that same batch. The old initial-only global match remains only as an audit
record; this proxy match is not claimed to be exact full-parameter gradient
budget equality.

## Artifact compatibility

Canonical critic artifacts use schema version 2 and include the runner version,
config hash, run class, dataset identity, loaded transition count, dimensions,
and canonical critic seed. Pilot artifacts, old v4.1 artifacts, or artifacts from
another formal identity fail closed rather than being silently reused.


## Validation/test separation

Checkpoint selection and acceptance use validation predictive quality plus best/final advantage stability on the actor-training split. Test R²/Pearson are final-report-only and never select or admit a checkpoint.
