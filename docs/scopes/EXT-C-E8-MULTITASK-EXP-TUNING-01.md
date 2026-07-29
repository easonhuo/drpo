# Scope contract: EXT-C-E8-MULTITASK-EXP-TUNING-01

## Approved responsibility

Implement the nine-task development tuning pilot that evaluates seven exponential remoteness retentions against one Positive-only baseline per task.

The repository owner explicitly approved the new Python path:

```text
src/drpo/e8_multitask_exp_tuning.py
```

Its sole responsibility is to reuse qualified P0 banks, sources, task generators, and verifiers; create new train-only P0-task reference adapters after the tuning split is frozen; preserve Countdown's archived train/validation split and supplied reference; build the frozen 72-cell/5-wave plan; execute fixed-budget Exp and Positive-only cells; evaluate validation trajectories; select one Exp rho per task; and preserve leakage, terminal, and provenance audits.

The train-only references are a necessary anti-leakage successor input. They do not modify, overwrite, reinterpret, or replace the historical P0 100-update diagnostic adapters, which remain bound to P0 and are not used for tuning performance.

## Allowed files

- `src/drpo/e8_multitask_exp_tuning.py`
- `configs/e8_multitask_exp_tuning.yaml`
- `scripts/run_e8_multitask_exp_tuning.sh`
- `docs/experiments/EXT-C-E8-MULTITASK-EXP-TUNING-01.md`
- this scope record
- extensions to existing `tests/test_e8_multitask_p0.py`
- temporary validation/transport files that must be removed before final review

No other new Python path is approved.

## Scientific exclusions

This scope must not:

- alter P0 banks, task generators, verifiers, historical warm-start adapters, or P0 results;
- allow P0-task reference preparation to access validation or test prompts;
- modify the seven-point rho grid after observing results;
- add Linear, Quadratic, TOPR, AsymRE, Global, Hybrid, or SBRC training arms;
- use test partitions for tuning;
- declare convergence, significance, universal ranking, or categorical causal identification;
- merge, register, or launch the pilot automatically.

D-U1 remains categorical causal authority. Task performance, valid/structure diagnostics, and NaN/Inf numerical failures remain separate reports.
