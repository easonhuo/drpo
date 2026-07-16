# E8 Paper-Aligned Lambda Round 1 — Operator Launch

Experiment ID: `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LAMBDA-ROUND1-0.5B-01`

Status: registered development pilot; not yet run.

## Default launch

From the repository root:

```bash
bash scripts/run_countdown_e8_paper_aligned_lambda_one_click.sh
```

The launcher defaults to exactly **2 auto-selected eligible GPU slots**, with one cell process per selected GPU. The candidate pool is configured separately from the slot count; device selection uses the registered free-memory and utilization gates. An explicit operator override may change scheduling capacity, but it must not change formulas, seeds, data, cell identities, or the registered 18-cell matrix.

## Resume

```bash
bash scripts/run_countdown_e8_paper_aligned_lambda_resume_one_click.sh
```

Resume is identity-checked against the RunSpec, configuration, calibration artifact, and existing cell manifests. Completed matching cells are retained; mismatched cells fail closed rather than being silently reused.

## Scientific boundary

This first round localizes the useful `lambda` region under the paper-aligned formula

```text
D = negative mean completion-token log probability
z = relu((D - tau) / scale_c)
w = alpha * exp(-lambda * z)
```

with `alpha=1`, frozen calibrated `tau` and `scale_c`, no extra surprisal square, and no test-set parameter selection. It is a development pilot, not a formal method ranking.