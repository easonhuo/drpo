# C-U1-E4-CONV-01 terminal audit and closure decision

Original pre-registered 18/20 scientific acceptance: **False**
User-confirmed scoped E4 scientific closure (2026-06-26): **True**
Final scoped scientific status: **long-run validated**

The normalized full-data residual is retained as a diagnostic and is not a hard gate.
Positive-only is not rerun; its complete terminal dynamics remain assigned to E2.
No threshold, seed label, horizon, optimizer, learning rate, or numerical result was changed.

| alpha | expected state | expected / total | inconclusive | explicit opposite | original 18/20 gate | user-accepted scoped closure |
|---:|---|---:|---:|---:|---|---|
| 0.75 | stable_beneficial_extrapolation | 15/20 | 5 | 0 | False | True |
| 1.00 | stable_beneficial_extrapolation | 16/20 | 4 | 0 | False | True |
| 1.25 | stable_over_extrapolation | 15/20 | 5 | 0 | False | True |

All 60 scientific roles were unchanged from step 2000 to step 4000. Task-performance collapse, support/variance-boundary events, and NaN/Inf were each 0/60 in this convergence-resolution run.

The closure certifies the E4 long-horizon phase interpretation. It does not certify 20/20 fixed points and does not retroactively pass the original 18/20 gate.
