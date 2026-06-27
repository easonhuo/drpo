# D-U1 E6 minimum-change semantic-gap long-run

`D-U1-E6-SEMANTIC-GAP-LONGRUN-01` is a registered successor to
`D-U1-E6-SEMANTIC-LONGRUN-01`. It does not overwrite the completed dense E6 or
the separate group-based `D-U1-E6-CONDITIONAL-GAP-01` stress diagnostic.

## Scientific change budget

The successor reuses the original E6 implementation, 64-action semantic
catalogue, reward geometry, policy, fixed advantages, Adam optimizer, batch
size, and event definitions. Its only environment intervention is a logged-role
coverage mask:

- train and test contexts remain independent draws from the same distribution;
- exactly half of contexts are designated gap contexts by a deterministic state
  partition;
- on each gap context, the 16 actions with highest reward similarity are removed
  from positive, local-negative, and far-negative logged roles;
- the full reward oracle and evaluation action catalogue remain available;
- every action must still appear somewhere in the global log.

This is same-distribution held-out-context generalization with a structured
state-action support gap. It is not state-distribution OOD generalization.

## Registered utility curve

The formal method domain is exactly:

```text
alpha = 0.00, 0.25, 0.50, 0.75, 1.00
```

`alpha=0` is Positive-only. `alpha=1` retains the original negative gradient
without suppression. Values above one are excluded; they are negative-gradient
overweight stress conditions and are not part of the method claim.

The primary metric is overall expected semantic reward. Hidden-action
probability and support diagnostics remain explanatory only. Registered reward
checkpoints are 4000, 8000, 16000, 24000, and 32000 steps.

## Development evidence boundary

Temporary sandbox runs used development seeds 900--909 outside the repository
formal registry. They motivated the registered full legal alpha grid and the
long horizon, but they are not formal evidence and cannot be aggregated with the
formal seeds. The formal run uses untouched seeds 150--169.

If terminal windows do not establish a plateau, the result remains finite-step
evidence and no steady-state method ranking is allowed.

## Validation and launch

Validate the frozen configuration without consuming formal seeds:

```bash
PYTHONPATH=src python3 src/drpo/du1_e6_semantic_gap_longrun.py \
  --config configs/du1_e6_semantic_gap_longrun.yaml \
  --output-root /tmp/du1-e6-semantic-gap-check-unused \
  --check-only
```

After applying and committing the update on a clean `main` checkout, launch the
formal run through the registered hardened wrapper:

```bash
python3 scripts/run_du1_e6_semantic_gap_longrun.py \
  --work-dir "$HOME/DRPO_RUNS/D-U1-E6-SEMANTIC-GAP-LONGRUN-01/run_001" \
  --device cpu
```

The wrapper requires a clean worktree and an `origin/main` match, records the
exact commit, writes five-seed recovery checkpoints, and packages the raw result
through the canonical hardened artifact channel.
